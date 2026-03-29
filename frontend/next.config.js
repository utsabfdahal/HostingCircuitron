/** @type {import('next').NextConfig} */
const nextConfig = {
  async rewrites() {
    return [
      {
        source: "/circuit/:path*",
        destination: "/circuitjs/:path*",
      },
    ];
  },
};
module.exports = nextConfig;
